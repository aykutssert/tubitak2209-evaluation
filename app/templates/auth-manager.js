// auth-manager.js - Tüm sayfalarda kullanılacak ortak auth yöneticisi

import { initializeApp } from 'https://www.gstatic.com/firebasejs/9.0.0/firebase-app.js';
import { getAuth, onAuthStateChanged, signOut } from 'https://www.gstatic.com/firebasejs/9.0.0/firebase-auth.js';

// Firebase config
const firebaseConfig = {
    // Firebase config bilgileriniz
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const auth = getAuth(app);

class AuthManager {
    constructor() {
        this.currentUser = null;
        this.init();
    }

    init() {
        // Auth state değişikliklerini dinle
        onAuthStateChanged(auth, (user) => {
            this.currentUser = user;
            this.updateNavbar();
            this.handleAuthRedirect();
        });
    }

    updateNavbar() {
        const usernameElement = document.getElementById('navUsername') || document.getElementById('displayUsername');
        const logoutBtn = document.querySelector('.logout-btn');

        if (this.currentUser) {
            // Kullanıcı giriş yapmış
            const displayName = this.currentUser.displayName || 
                              this.currentUser.email?.split('@')[0] || 
                              'User';
            
            if (usernameElement) {
                usernameElement.textContent = displayName;
            }

            if (logoutBtn) {
                logoutBtn.textContent = 'Logout';
                logoutBtn.onclick = () => this.logout();
            }

            // localStorage'a da kaydet (opsiyonel)
            localStorage.setItem('username', displayName);
            localStorage.setItem('userEmail', this.currentUser.email);
            
        } else {
            // Kullanıcı giriş yapmamış
            if (usernameElement) {
                usernameElement.textContent = 'Guest';
            }
        }
    }

    handleAuthRedirect() {
        const currentPath = window.location.pathname;
        const publicPages = ['/login', '/register', '/login.html', '/register.html'];
        
        if (this.currentUser) {
            // Kullanıcı giriş yapmış, login/register sayfasındaysa home'a yönlendir
            if (publicPages.some(page => currentPath.includes(page))) {
                window.location.href = '/';
            }
        } else {
            // Kullanıcı giriş yapmamış, korumalı sayfadaysa login'e yönlendir
            if (!publicPages.some(page => currentPath.includes(page))) {
                window.location.href = '/login';
            }
        }
    }

    async logout() {
        try {
            await signOut(auth);
            localStorage.clear(); // Tüm local storage'ı temizle
            window.location.href = '/login';
        } catch (error) {
            console.error('Logout error:', error);
            alert('Çıkış yapılırken hata oluştu');
        }
    }

    // Kullanıcının giriş durumunu kontrol et
    isLoggedIn() {
        return this.currentUser !== null;
    }

    // Kullanıcı bilgilerini al
    getCurrentUser() {
        return this.currentUser;
    }
}

// Global auth manager instance
window.authManager = new AuthManager();

export default AuthManager;